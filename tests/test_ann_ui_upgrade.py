"""标注平台测试：4 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_ui_upgrade
- test_v06_b3_fixes
- test_v06_b4_fixes
- test_batch11_ui
"""


# == 来源模块：test_ui_upgrade ==
"""UI 升级测试：Token 化主题 + 公共组件 + 回归锁。

约束：（字号≥15px、模板占位
全部替换、Cython 槽包装）；组件行为与 ui_widgets/ui_style 对应。
"""

import re
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from PyQt5.QtWidgets import QLabel

from xccv.ui_widgets import load_prefs, save_prefs


class ThemeTokenTest(unittest.TestCase):
    """期0：ui_style Token 化（主色 #5ec6dd）。"""

    def test_app_qss_generated_from_tokens(self):
        from xccv.ui_style import APP_QSS, build_qss

        self.assertEqual(APP_QSS, build_qss("light"))
        self.assertIn("#5ec6dd", APP_QSS)          # 主色定稿 D2
        self.assertNotIn("@", APP_QSS)             # 模板占位必须全部替换

    def test_font_floor_15px(self):
        from xccv.ui_style import APP_QSS

        sizes = [int(m) for m in re.findall(r"font-size:\s*(\d+)px", APP_QSS)]
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), 15, f"过小字号：{sizes}")

    def test_unknown_theme_rejected(self):
        from xccv.ui_style import build_qss

        with self.assertRaises(ValueError):
            build_qss("bogus")   # 仅 light/dark 两种

    def test_dark_theme_builds(self):
        from xccv.ui_style import build_qss

        dark = build_qss("dark")
        self.assertIn("#1c2634", dark)             # 暗色底
        self.assertIn("#5ec6dd", dark)             # 主色不变
        self.assertNotIn("#1c2634", build_qss("light"))

    def test_version_bumped(self):
        import xccv

        self.assertEqual(xccv.__version__, "0.8.3")

    def test_launcher_has_deploy_family(self):
        from xccv.launcher import PLATFORMS

        for key in ("deploy", "model_load", "csharp_test"):
            self.assertIn(key, PLATFORMS)
            self.assertEqual(PLATFORMS[key][1], "xccv.deploy_platform")
        self.assertEqual(PLATFORMS["deploy"][0], "XCcv推理服务.exe")

    def test_canvas_page_recolored(self):
        html = (Path(__file__).resolve().parents[1]
                / "xccv" / "modeltools" / "web" / "index.html") \
            .read_text(encoding="utf-8")
        for old in ("6d28d9", "2563eb", "60a5fa"):
            self.assertNotIn(old, html)

    def test_legacy_aliases_point_to_tokens(self):
        from xccv import ui_style

        self.assertEqual(ui_style.ACCENT, ui_style.C.primary_deep)
        self.assertEqual(ui_style.BG, ui_style.C.surface)
        self.assertEqual(ui_style.ACCENT2, ui_style.C.accent2)

    def test_btnaccent_rule_exists(self):
        """部署页按钮 objectName=btnAccent 曾无对应 QSS 规则（潜伏缺口）。"""
        from xccv.ui_style import APP_QSS

        self.assertIn("QPushButton#accent, QPushButton#btnAccent", APP_QSS)


class UiWidgetsTest(unittest.TestCase):
    """期0：公共组件（Badge/StatusDot/Toast/EmptyState/工具函数）。"""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_badge_kinds(self):
        from xccv.ui_widgets import Badge

        self.assertEqual(Badge("检测", "orange").objectName(), "badgeDet")
        self.assertEqual(Badge("运行中", "success").objectName(), "badgeSeg")
        self.assertEqual(Badge("x", "未知kind").objectName(), "badge")  # 兜底

    def test_status_dot_breath(self):
        from xccv.ui_widgets import StatusDot

        dot = StatusDot()
        self.assertFalse(dot.breathing())
        dot.start_breath()
        self.assertTrue(dot.breathing())
        dot._on_breath(0.25)   # 三角波中段：透明度明显高于下限
        self.assertGreater(dot._alpha, 0.35)
        dot.stop_breath()
        self.assertFalse(dot.breathing())
        self.assertEqual(dot._alpha, 1.0)

    def test_toast_single_active_and_close(self):
        from PyQt5.QtWidgets import QWidget

        from xccv.ui_widgets import toast

        host = QWidget()
        t1 = toast(host, "已保存", "success")
        self.assertEqual(t1.objectName(), "toastSuccess")
        self.assertIs(host._active_toast, t1)
        t2 = toast(host, "覆盖旧条", "warn")   # 同窗新 Toast 让旧条即刻让位
        self.assertTrue(t1._closing)
        self.assertIs(host._active_toast, t2)
        t2.close_now()

    def test_toast_stale_ref_after_autodismiss(self):
        """Toast 自动淡出销毁后窗口引用必须摘除。

        回归背景：淡出销毁未清 _active_toast，窗口持有已删 C++ 对象的
        Python 壳；下一条 toast 调 old.close_now() → self._anim.stop()
        抛 RuntimeError，训练平台保存参数/开始训练等按钮全部失效。
        """
        from PyQt5.QtCore import QEvent
        from PyQt5.QtWidgets import QApplication, QWidget

        from xccv.ui_widgets import toast

        app = QApplication.instance() or QApplication([])
        host = QWidget()
        t1 = toast(host, "第一条", "info")
        t1._closing = True
        t1._on_fade_done()                       # 淡出完成：先摘引用再销毁
        self.assertIsNone(host._active_toast)
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)  # 立即执行 deleteLater
        t1.close_now()                           # 死壳重复关闭不再抛 RuntimeError
        t1._play_out()                           # 停留定时器晚于销毁的竞态：静默返回
        t2 = toast(host, "第二条", "success")    # 修复前：此处被死引用截断
        self.assertIs(host._active_toast, t2)
        t2._closing = True
        t2._on_fade_done()
        self.assertIsNone(host._active_toast)
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        t2.close_now()

    def test_empty_state_rows(self):
        from xccv.ui_widgets import EmptyState

        full = EmptyState("🗂", "标题", "提示")
        self.assertEqual(full.layout().count(), 3)
        no_hint = EmptyState("🗂", "标题")
        self.assertEqual(no_hint.layout().count(), 2)

    def test_relative_time(self):
        from xccv.ui_widgets import relative_time

        self.assertEqual(relative_time(datetime.now() - timedelta(minutes=5)),
                         "5 分钟前")
        self.assertEqual(relative_time(datetime.now() - timedelta(hours=3)),
                         "3 小时前")
        self.assertEqual(relative_time("不是日期"), "不是日期")  # 兜底原样返回

    def test_nav_icons_render(self):
        from xccv.ui_widgets import nav_icon

        for kind in ("ai", "trad", "flow", "deploy", "settings", "doc", "about",
                     "unknown-fallback"):
            pm = nav_icon(kind, "#5f7086").pixmap(40, 40)
            self.assertEqual(pm.width(), 40)

    def test_module_has_slot_wrapper(self):
        src = (Path(__file__).resolve().parents[1] / "xccv" / "ui_widgets.py") \
            .read_text(encoding="utf-8")
        self.assertIn("apply_auto_slot_module(globals())", src)


class ServiceHealthPollerTest(unittest.TestCase):
    """期1：共享健康轮询三态分类（running / starting / stopped）。"""

    @classmethod
    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _poller(self):
        from xccv.service_health import ServiceHealthPoller

        return ServiceHealthPoller(interval_ms=10 ** 6)  # 不真启动定时器

    def test_states_published(self):
        poller = self._poller()
        seen = []
        poller.state_changed.connect(lambda s, i: seen.append((s, i)))
        poller.note_starting()                       # 部署页拉起进程 → 立即启动中
        self.assertEqual(seen[-1][0], "starting")
        poller._publish("running", {"loaded_models": 2})
        self.assertEqual(seen[-1][0], "running")
        self.assertEqual(seen[-1][1]["models"], 2)

    def test_failure_classification_window(self):
        poller = self._poller()
        import time

        poller.note_starting()
        self.assertEqual(poller._classify_failure(), "starting")  # 窗口期内
        poller._starting_since = time.time() - 60                 # 超窗 → 停止
        self.assertEqual(poller._classify_failure(), "stopped")
        poller._starting_since = 0.0
        self.assertEqual(poller._classify_failure(), "stopped")   # 从未启动过

    def test_state_field_tracks_last(self):
        poller = self._poller()
        poller._publish("running", {})
        self.assertEqual(poller.state(), "running")
        poller._publish("stopped", {})
        self.assertEqual(poller.state(), "stopped")


class RecentProjectsTest(unittest.TestCase):
    """期1：最近项目直读（轻量、不依赖 training.project 导入链）。"""

    def test_reads_sorted_by_mtime(self):
        import json as _json
        import tempfile

        from xccv.app import recent_projects

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, offset in (("新项目", 100.0), ("旧项目", 50.0)):
                d = root / name
                d.mkdir()
                (d / "project.json").write_text(
                    _json.dumps({"name": name}, ensure_ascii=False),
                    encoding="utf-8")
                import os

                t = 1_700_000_000 + offset
                os.utime(d / "project.json", (t, t))
            (root / "杂鱼目录").mkdir()  # 无 project.json，应被忽略
            rows = recent_projects(root=root)
            self.assertEqual([r["name"] for r in rows], ["新项目", "旧项目"])


class HomeWindowShellTest(unittest.TestCase):
    """期1：主界面骨架（图标导航 / 顶栏 / 搜索过滤 / 状态胶囊）。"""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        from xccv.app import HomeWindow

        win = HomeWindow()
        self.addCleanup(win.close)
        return win

    def test_nav_and_topbar(self):
        win = self._window()
        self.assertEqual(win.nav_list.count(), 4)
        self.assertTrue(hasattr(win, "service_chip"))
        self.assertTrue(hasattr(win, "search_edit"))
        self.assertTrue(hasattr(win, "page_title"))

    def test_nav_switch_updates_title(self):
        win = self._window()
        win.nav_list.setCurrentRow(3)   # 部署平台：普通切页（卡片自行拉起进程）
        self.assertEqual(win.page_title.text(), "部署平台")

    def test_search_filters_cards(self):
        win = self._window()
        win._apply_search("训练")
        visible = [c for c in win._platform_cards if not c.isHidden()]
        self.assertEqual([c.card_key for c in visible], ["training"])
        win._apply_search("")
        visible = [c for c in win._platform_cards if not c.isHidden()]
        self.assertEqual(len(visible), 4)

    def test_recent_projects_refresh(self):
        """最近项目随主界面显示/回 AI 页/窗口激活实时重建。"""
        from unittest import mock

        from xccv import app as app_mod

        win = self._window()
        fake = [{"name": "项目甲", "mtime": 1_700_000_100},
                {"name": "项目乙", "mtime": 1_700_000_000}]
        with mock.patch.object(app_mod, "recent_projects", return_value=fake):
            win._refresh_recent_projects()
            lay = win._recent_chips_lay
            names = [c.project_name for c in win._recent_chips]
            self.assertEqual(names, ["项目甲", "项目乙"])
            self.assertEqual(lay.count(), 3)          # 2 卡 + 尾部弹簧
            self.assertFalse(win.recent_row.isHidden())
            # 数据变化后再次刷新：旧卡清空不残留
            win._refresh_recent_projects()
            self.assertEqual(len(win._recent_chips), 2)
            self.assertEqual(lay.count(), 3)
            #11697旧卡必须立即脱离视图（仅 deleteLater 会残影叠字）
            stale = win._recent_chips[0]
            win._refresh_recent_projects()
            self.assertIsNone(stale.parent())
            # 回 AI 平台页（导航 row 0）触发刷新（经 row 2，避免拉起工作台）
            with mock.patch.object(app_mod, "recent_projects",
                                   return_value=fake[:1]) as rp:
                win.nav_list.setCurrentRow(2)
                win.nav_list.setCurrentRow(0)
                rp.assert_called()
            self.assertEqual([c.project_name for c in win._recent_chips],
                             ["项目甲"])
            # 窗口激活变化（从训练平台回来）触发刷新：changeEvent 收到的是
            # ActivationChange（WindowActivate 不进 changeEvent）
            from PyQt5.QtCore import QEvent

            with mock.patch.object(app_mod, "recent_projects",
                                   return_value=fake) as rp:
                win.event(QEvent(QEvent.ActivationChange))
                rp.assert_called()
        self.addCleanup(win._refresh_recent_projects)  # 清理桩数据影响

    def test_nav_icon_sizes_enlarged(self):
        """v0.5.5追加-④二轮：导航图标 45/字号 17px，底区按钮图标 36。"""
        from PyQt5.QtWidgets import QPushButton

        from xccv.ui_style import APP_QSS

        win = self._window()
        self.assertEqual(win.nav_list.iconSize().width(), 45)
        for btn in win.findChildren(QPushButton):
            if btn.objectName() == "navFoot":
                self.assertEqual(btn.iconSize().width(), 36)
        m = re.search(r"QListWidget#navList \{[^}]*\}", APP_QSS)
        self.assertIsNotNone(m)
        self.assertIn("font-size: 17px", m.group(0))

    def test_search_mode_and_page_dispatch(self):
        """v0.5.5追加修：搜索态隐藏 Hero；最近项目【保留搜索】（需求：
        更正——上轮只该修残影 bug，不该砍功能），命中卡作为结果显示。"""
        from unittest import mock

        from xccv import app as app_mod

        win = self._window()
        fake = [{"name": "项目甲", "mtime": 1_700_000_000},
                {"name": "项目乙", "mtime": 1_700_000_001}]
        with mock.patch.object(app_mod, "recent_projects", return_value=fake):
            win._refresh_recent_projects()
            # 搜索"项目甲"：Hero 隐藏；最近项目节可见且只显示命中卡
            win._apply_search("项目甲")
            self.assertTrue(win.hero_row.isHidden())
            self.assertFalse(win.recent_row.isHidden())
            shows = [c.project_name for c in win._recent_chips
                     if not c.isHidden()]
            self.assertEqual(shows, ["项目甲"])
            # 无命中：最近项目节隐藏（不显示空节）
            win._apply_search("不存在的关键词")
            self.assertTrue(win.recent_row.isHidden())
            # 清空恢复全部
            win._apply_search("")
            self.assertFalse(win.recent_row.isHidden())
            self.assertEqual(len([c for c in win._recent_chips
                                  if not c.isHidden()]), 2)
            self.assertFalse(win.hero_row.isHidden())

    def test_search_dispatch_to_pages(self):
        """部署页/流程页接入全局搜索（每页可搜本页内容）。"""
        from xccv.deploy_platform import DeployPage
        from xccv.flow_platform import FlowPage

        dep = DeployPage()
        self.assertEqual(len(dep._cards), 3)
        dep.set_search_query("c#")
        vis = [c.card_key for c in dep._cards if not c.isHidden()]
        self.assertEqual(vis, ["csharp_test"])
        dep.set_search_query("")
        self.assertEqual(len([c for c in dep._cards if not c.isHidden()]), 3)

        fp = FlowPage()
        fp.set_search_query("img00")   # 与本页搜索框同步并过滤图片树
        self.assertEqual(fp.search_edit.text(), "img00")
        fp.set_search_query("")
        self.assertEqual(fp.search_edit.text(), "")

        # 传统平台页：首次导航替换为算子工作台（非占位）——搜索过滤算子列表
        from xccv.workbench import OperatorWorkbench

        wb = OperatorWorkbench()
        total = wb.op_list.count()
        self.assertGreater(total, 0)
        wb.set_search_query("blob")
        self.assertGreaterEqual(wb.op_list.count(), 1)
        self.assertLess(wb.op_list.count(), total)
        self.assertTrue(all("blob" in wb.op_list.item(i).text().lower()
                            for i in range(wb.op_list.count())))
        wb.set_search_query("")
        self.assertEqual(wb.op_list.count(), total)

    def test_service_chip_states(self):
        win = self._window()
        chip = win.service_chip
        chip.set_state("running", 2)
        self.assertIn("2 个模型", chip.text.text())
        chip.set_state("starting", 0)
        self.assertTrue(chip.dot.breathing())
        chip.set_state("stopped", 0)
        self.assertFalse(chip.dot.breathing())

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _poller(self):
        from PyQt5.QtCore import QObject, pyqtSignal

        class _FakePoller(QObject):
            state_changed = pyqtSignal(str, dict)

            def state(self):
                return "stopped"

        return _FakePoller()

    def _island(self, poller):
        from xccv.island import ServiceIsland

        island = ServiceIsland(poller)
        self.addCleanup(island.close)
        return island

    def test_three_states(self):
        poller = self._poller()
        island = self._island(poller)
        poller.state_changed.emit("starting", {"url": "u", "models": 0})
        self.assertEqual(island.cap_label.text(), "推理服务启动中…")
        self.assertTrue(island.dot.breathing())
        poller.state_changed.emit("running", {"url": "http://x:1", "models": 2})
        self.assertEqual(island.cap_label.text(), "推理服务 · 2 个模型")
        self.assertFalse(island.dot.breathing())
        poller.state_changed.emit("stopped", {"url": "", "models": 0})
        self.assertEqual(island.cap_label.text(), "推理服务未运行")
        self.assertEqual(island.state(), "stopped")

    def test_expand_toggle_and_fill(self):
        poller = self._poller()
        island = self._island(poller)
        poller.state_changed.emit("running", {"url": "http://x:1", "models": 1})
        island._expanded = True
        island.panel.setVisible(True)
        island._on_expand_step(118.0)
        self.assertGreater(island.height(), 40)
        island._fill_details({"models": [
            {"model_id": "斑点分割-1", "alias": "斑点", "task": "seg"}]})
        self.assertIn("斑点", island.models_label.text())

    def test_hover_expands_and_leave_collapses(self):
        """验收 2-2 交互修正：悬停自动展开，移走自动收起。"""
        from PyQt5.QtCore import QEvent

        poller = self._poller()
        island = self._island(poller)
        poller.state_changed.emit("stopped", {"url": "", "models": 0})
        island.enterEvent(QEvent(QEvent.Enter))
        self.assertTrue(island._expanded)
        island.leaveEvent(QEvent(QEvent.Leave))
        island._on_expand_done()   # 动画立即结束（测试不等待 220ms）
        self.assertFalse(island._expanded)
        self.assertFalse(island.panel.isVisibleTo(island))

    def test_button_texts_shortened(self):
        """验收 2-2：「打开部署平台」显示不全 → 缩短为「部署平台」。"""
        poller = self._poller()
        island = self._island(poller)
        self.assertEqual(island.btn_open.text(), "部署平台")
        self.assertEqual(island.btn_key.text(), "复制 Key")

    def test_island_module_has_slot_wrapper(self):
        src = (Path(__file__).resolve().parents[1] / "xccv" / "island.py") \
            .read_text(encoding="utf-8")
        self.assertIn("apply_auto_slot_module(globals())", src)

    def test_deploy_page_island_toggle(self):
        from unittest import mock

        from xccv.deploy_platform import ServicePanel

        poller = self._poller()
        mem: dict = {}
        with mock.patch("xccv.ui_widgets.load_prefs", lambda: dict(mem)), \
                mock.patch("xccv.ui_widgets.save_prefs",
                           lambda d: mem.clear() or mem.update(d)):
            page = ServicePanel(poller=poller)
            self.addCleanup(page.close)
            # 默认开（prefs 无记录时）：悬浮窗实例已创建
            self.assertIsNotNone(page._island)
            page.island_check.setChecked(False)
            self.assertIsNone(page._island)
            self.assertIn("island_enabled", mem)
            page.island_check.setChecked(True)
            self.assertIsNotNone(page._island)

    def test_deploy_page_no_poller_no_island(self):
        from xccv.deploy_platform import ServicePanel

        page = ServicePanel()
        self.addCleanup(page.close)
        self.assertIsNone(page._island)   # 独立构造不产悬浮窗
        self.assertTrue(page.island_check.isVisibleTo(page)
                        or page.island_check.isHidden() is False)


class _TrainEnv(unittest.TestCase):
    """期3 基座：临时项目目录 + 一个 4 图检测项目（复用 batch11 模式）。"""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import tempfile
        from unittest import mock

        from tests import synthetic
        from xccv.training import project as project_mod

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._prefs_backup = dict(load_prefs())
        self.addCleanup(save_prefs, self._prefs_backup)
        # 隔离用户真实偏好（如 train_param_mode=expert 会改变默认页）。
        # 注意 save_prefs 是合并写（读旧文件 update 后写回）——合并删不掉
        # 文件里已有的键，必须整写文件才能隔离
        clean = dict(self._prefs_backup)
        clean.pop("train_param_mode", None)
        import json as _json

        from xccv.ui_widgets import _prefs_path

        _pp = _prefs_path()
        _pp.parent.mkdir(parents=True, exist_ok=True)
        _pp.write_text(_json.dumps(clean, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        det, _ = synthetic.make_labelme_dataset(Path(self._tmp.name) / "det",
                                                n_images=4)
        from xccv.training.project import create_project

        create_project("ui", str(det), "", "detection")

    def _window(self):
        from xccv.training.project import load_project
        from xccv.training.training_window import ProjectWindow

        pw = ProjectWindow(load_project("ui"))
        self.addCleanup(pw.close)
        return pw


class QuickParamPanelTest(_TrainEnv):
    """期3：Quick/专家双轨 + 预设卡 + 推荐徽章 + 恢复默认。"""

    def test_default_quick_mode_visible(self):
        pw = self._window()
        self.assertFalse(pw.quick_panel.isHidden())
        self.assertTrue(pw.expert_host.isHidden())

    def test_mode_switch_and_sync(self):
        pw = self._window()
        pw.q_epochs.setValue(55)
        pw._experts_from_quick()
        self.assertEqual(pw.par_epochs.value(), 55)     # quick → experts
        pw.par_epochs.setValue(77)
        pw._quick_from_experts()
        self.assertEqual(pw.q_epochs.value(), 77)       # experts → quick
        pw._set_param_mode(False)
        self.assertTrue(pw.quick_panel.isHidden())
        self.assertFalse(pw.expert_host.isHidden())
        self.assertFalse(pw.btn_expand_all.isHidden())
        pw._set_param_mode(True)
        self.assertFalse(pw.quick_panel.isHidden())
        self.assertTrue(pw.expert_host.isHidden())

    def test_preset_apply_updates_params_and_badges(self):
        pw = self._window()
        base = pw.par_epochs.value()
        pw._apply_preset("fast")
        self.assertEqual(pw.par_epochs.value(), 30)
        self.assertTrue(pw.par_freeze.isChecked())
        self.assertTrue(pw._rec_badges["epochs"].isVisible()
                        or not pw._rec_badges["epochs"].isHidden())
        self.assertGreaterEqual(pw.project.epochs, 1)
        self.assertEqual(pw.par_epochs.value(), 30)   # 保存回读一致
        del base

    def test_preset_apply_qc_quality(self):
        pw = self._window()
        pw._apply_preset("quality")
        self.assertEqual(pw.par_epochs.value(), 300)
        self.assertTrue(pw.par_tf32.isChecked())
        # 4 图小数据集会触发 recommend_settings 的「冻结骨干」建议：
        # 预设联动推荐后 freeze 应为 True（联动语义覆盖预设初值）
        self.assertTrue(pw.par_freeze.isChecked())

    def test_reset_defaults(self):
        pw = self._window()
        before = pw.par_epochs.value()
        pw._apply_preset("fast")
        pw._reset_quick_defaults()
        self.assertEqual(pw.par_epochs.value(), before)
        self.assertTrue(pw._rec_badges["epochs"].isHidden())

    def test_save_params_syncs_from_quick(self):
        pw = self._window()
        pw.q_batch.setValue(9)
        pw._save_params()
        self.assertEqual(pw.par_batch.value(), 9)
        self.assertEqual(pw.project.batch_size, 9)

    def test_preset_values_registered(self):
        pw = self._window()
        self.assertEqual(set(pw._preset_values), {"fast", "balanced", "quality"})
        # V0.7.4 一.1：免训练快速模式新增 k_shot/aug_count 徽章行
        self.assertEqual(set(pw._rec_badges),
                         {"epochs", "batch", "lr", "size",
                          "k_shot", "aug_count"})


class Phase4PolishTest(_TrainEnv):
    """期4：项目卡片墙优化（悬停卡/相对时间）、训练历史状态点、Toast。"""

    def test_project_card_is_hover_card_with_relative_time(self):
        from xccv.training.training_window import TrainingWindow, _ProjectCard

        win = TrainingWindow()
        self.addCleanup(win.close)
        self.assertGreater(win.project_count(), 0)
        card = win._cards[0]
        self.assertIsInstance(card, _ProjectCard)
        self.assertTrue(hasattr(card, "_shadow"))   # HoverCard 悬停阴影
        labels = card.findChildren(QLabel)
        self.assertTrue(any(l.text().startswith("更新于") for l in labels))

    def test_run_item_has_status_dot(self):
        from xccv.training.runs import RunRecord
        from xccv.training.training_window import _status_dot_icon

        pm = _status_dot_icon("#e45756")
        self.assertEqual(pm.width(), 14)
        pw = self._window()
        rec = RunRecord(run_dir=Path(pw.project.model_dir)
                        / "20260920_100000_yolo11n",
                        status="failed", start_time="2026-09-20 10:00:00",
                        epochs_done=3, epochs_planned=30)
        item = pw._make_run_item(rec)
        self.assertFalse(item.icon().isNull())
        self.assertIn("失败", item.text())
        self.assertIn("yolo11n", item.text())

    def test_toast_replaces_statusbar_on_save(self):
        pw = self._window()
        pw._save_params()
        active = getattr(pw.window(), "_active_toast", None)
        self.assertIsNotNone(active)
        self.assertEqual(active.objectName(), "toastSuccess")
        active.close_now()


class ChromeHitTestTest(unittest.TestCase):
    """无边框命中测试（QPoint 漏导入导致鼠标移动
    即 NameError；此处直接构造 WM_NCHITTEST 消息走 nativeEvent）。"""

    @classmethod
    def setUpClass(cls):
        import os
        from unittest import mock

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication, QSystemTrayIcon

        #27157offscreen 无托盘——ServiceWindow.closeEvent 的"无托盘=退出
        # 应用"分支会调 QApplication.quit() 毒化本进程全部后续 timer（表现
        # 为后续用例 timer 全灭、套件尾部段错误/自旋）。测试环境声明托盘可用。
        cls._tray_patch = mock.patch.object(
            QSystemTrayIcon, "isSystemTrayAvailable", return_value=True)
        cls._tray_patch.start()
        cls.addClassCleanup(cls._tray_patch.stop)
        cls.app = QApplication.instance() or QApplication([])

    def _hit(self, win, x, y):
        import ctypes
        import ctypes.wintypes

        from PyQt5.QtCore import QPoint

        from xccv.win_chrome import _HT, _HT_CAPTION

        # WM_NCHITTEST 的 lParam 是全局屏幕坐标；测试传入本地坐标先换算，
        # 避免依赖窗口恰好位于原点（offscreen 多窗口级联会平移位置）
        gp = win.mapToGlobal(QPoint(x, y))
        msg = ctypes.wintypes.MSG()
        msg.message = 0x0084
        msg.lParam = ((gp.y() & 0xFFFF) << 16) | (gp.x() & 0xFFFF)
        handled, result = win.nativeEvent(
            b"windows_generic_MSG", ctypes.addressof(msg))
        if not handled:
            return None
        for name, code in _HT.items():
            if result == code:
                return name
        if result == _HT_CAPTION:
            return "CAPTION"
        return str(result)

    def test_hit_zones(self):
        from xccv.deploy_platform import ServiceWindow

        win = ServiceWindow()
        win.show()
        self.addCleanup(win.close)
        app = self.app
        app.processEvents()
        w, h = win.width(), win.height()
        self.assertIsNone(self._hit(win, w // 2, h // 2))     # 中央 → 交回 Qt
        self.assertEqual(self._hit(win, 2, h // 2), "L")      # 左缘
        self.assertEqual(self._hit(win, w - 2, h // 2), "R")  # 右缘
        self.assertEqual(self._hit(win, w // 2, 2), "T")      # 上缘
        self.assertEqual(self._hit(win, w // 2, h - 2), "B")  # 下缘
        self.assertEqual(self._hit(win, 2, 2), "TL")
        self.assertEqual(self._hit(win, w - 2, h - 2), "BR")
        # 顶栏空白/标题 → 可拖（HTCAPTION）
        self.assertEqual(self._hit(win, w // 2, 19), "CAPTION")
        # 顶栏按钮区 → 不拦截（交回 Qt，按钮可点）
        self.assertIsNone(self._hit(win, w - 21, 19))

    def test_qpoint_import_present(self):
        """源码锁：nativeEvent 依赖 QPoint，重写模块不得漏导入。"""
        src = (Path(__file__).resolve().parents[1] / "xccv"
               / "win_chrome.py").read_text(encoding="utf-8")
        self.assertIn("QPoint", src.split("def nativeEvent")[0])


class TaskbarAumidTest(unittest.TestCase):
    """v0.5.5：任务栏按大平台分组（进程 AUMID）。"""

    def test_aumid_mapping(self):
        from xccv.taskbar import aumid_for

        self.assertEqual(aumid_for("deploy"), "XCcv.InferenceService")
        self.assertEqual(aumid_for("model_load"), "XCcv.ModelLoading")
        self.assertEqual(aumid_for("csharp_test"), "XCcv.CSharpTesting")
        self.assertEqual(aumid_for("training"), "XCcv.training")

    def test_set_returns_bool(self):
        from xccv.taskbar import set_process_app_user_model_id

        self.assertIsInstance(set_process_app_user_model_id("XCcv.Test"), bool)

    def test_platform_entry_wired(self):
        src = (Path(__file__).resolve().parents[1] / "xccv"
               / "platform_entry.py").read_text(encoding="utf-8")
        self.assertIn("set_process_app_user_model_id(aumid_for(key))", src)
        app_src = (Path(__file__).resolve().parents[1] / "xccv"
                   / "app.py").read_text(encoding="utf-8")
        self.assertIn('set_process_app_user_model_id("XCcv.Main")', app_src)

    def test_chrome_windows_have_platform_icons(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication, QSystemTrayIcon

        app = QApplication.instance() or QApplication([])
        from unittest import mock

        from xccv.deploy_platform import (
            CSharpWindow, ModelServiceWindow, ServiceWindow)

        mem: dict = {}
        with mock.patch("xccv.ui_widgets.load_prefs", lambda: dict(mem)),                 mock.patch("xccv.ui_widgets.save_prefs",
                           lambda d: mem.clear() or mem.update(d)),                 mock.patch.object(
                QSystemTrayIcon, "isSystemTrayAvailable", return_value=True):
            #31479同 ChromeHitTestTest——close 也必须在补丁域内，否则
            # ServiceWindow.closeEvent 走"无托盘=quit()"毒化进程全部 timer
            sw = ServiceWindow()
            mw = ModelServiceWindow()
            cs = CSharpWindow()
            for w in (sw, mw, cs):
                self.assertFalse(w.windowIcon().isNull(),
                                 f"{w.windowTitle()} 缺窗口图标")
                w.close()


class ServiceWindowTest(unittest.TestCase):
    """v0.5.2：独立部署窗口（托盘 / 灵动岛快捷开关 / 进程独立）。"""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        from unittest import mock

        from xccv.deploy_platform import ServiceWindow

        mem: dict = {}
        with mock.patch("xccv.ui_widgets.load_prefs", lambda: dict(mem)), \
                mock.patch("xccv.ui_widgets.save_prefs",
                           lambda d: mem.clear() or mem.update(d)):
            win = ServiceWindow()
        self.addCleanup(win.hide)
        return win

    def test_window_contains_page_and_island(self):
        from PyQt5.QtCore import Qt

        from xccv.deploy_platform import ServicePanel, ServiceWindow

        win = self._window()
        self.assertIsInstance(win._page, ServicePanel)
        # v0.5.3-b：页面挂在基座 body_host（顶栏之下）
        self.assertIs(win._page.parent(), win._body)   # v0.5.5 壳体布局
        self.assertIsNotNone(win._page._island)   # 默认开启灵动岛
        self.assertEqual(win.windowTitle(), "XCcv 推理服务")
        # 顶栏三键在（v0.5.2 曾因覆盖 centralWidget 而丢失）
        from PyQt5.QtWidgets import QPushButton

        btns = [b for b in win.findChildren(QPushButton)
                if b.objectName().startswith("chrome")]
        self.assertEqual([b.objectName() for b in btns],
                         ["chromeMin", "chromeMax", "chromeClose"])
        # v0.5.5：三键统一尺寸 + 圆角透明壳
        self.assertEqual(len({(b.width(), b.height()) for b in btns}), 1)
        self.assertTrue(win.testAttribute(Qt.WA_TranslucentBackground))

    def test_tray_island_toggle(self):
        win = self._window()
        self.assertTrue(win._page.island_check.isChecked())
        win._toggle_island()
        self.assertFalse(win._page.island_check.isChecked())
        self.assertEqual(win._act_island.text(), "显示灵动岛")
        win._toggle_island()
        self.assertEqual(win._act_island.text(), "隐藏灵动岛")

    def test_restore_from_tray_signal(self):
        win = self._window()
        win.show()                       # 未 show 的顶层窗口本就报告 hidden
        self.assertFalse(win.isHidden())
        win.hide()
        win._page.open_deploy_requested.emit()   # 灵动岛按钮 → 激活窗口
        self.assertFalse(win.isHidden())


class StoppingStateMachineTest(unittest.TestCase):
    """Q1：停止中独立状态 + 提示闭环 + 停止失败提示。"""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _page(self):
        from xccv.deploy_platform import ServicePanel

        page = ServicePanel()
        self.addCleanup(page.close)
        return page

    def test_stop_enters_stopping_then_confirmed(self):
        page = self._page()
        page._stop_service()
        self.assertTrue(page._service_stopping)
        self.assertIn("正在停止", page.status_label.text())
        self.assertFalse(page.btn_toggle.isEnabled())
        page._apply_poll(False, True)   # 轮询确认已停止
        self.assertFalse(page._service_stopping)
        self.assertIn("已停止", page.tip_label.text())
        self.assertIn("未运行", page.status_label.text())
        self.assertTrue(page.btn_toggle.isEnabled())

    def test_stop_failed_when_still_running(self):
        page = self._page()
        page._service_stopping = True
        page._apply_poll(True, False)   # health 仍 200：停止失败
        self.assertFalse(page._service_stopping)
        self.assertIn("停止失败", page.tip_label.text())
        self.assertIn("运行中", page.status_label.text())

    def test_start_clears_stopping(self):
        page = self._page()
        page._service_stopping = True
        page._start_service()   # 端口未占用路径也会清 stopping
        self.assertFalse(page._service_stopping)

# == 来源模块：test_v06_b3_fixes ==
"""回归。

- 标注：L 键智能标注（NameError）/ 批量预标注（auto_cleanup_thread 缺参）/
  画笔局部重绘 / 双击判定 250ms / 切图「保持缩放」开关；
- 训练：小目标锚框全链路（UI→Project→TrainConfig→ckpt）/ 卡片按更新时间倒序；
- 模型工具：缺陷生成固定优先端口（同进程多实例不撞端口）/ 加速尺寸建议 + 进度条；
- YOLO（anchor-free）独立工业损失 NWD-CIoU / 宽松口径逐轮指标（validator 检出收集）。
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PyQt5 import QtCore, QtGui, QtWidgets

import xccv.annotation  # noqa: F401 先导入：固定 PyQt5 绑定并在需要时装 osam 桩
from xccv.ui_widgets import _prefs_path

ROOT = Path(__file__).resolve().parents[1]
_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def _pump_until(pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        QtWidgets.QApplication.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return False


class _PrefsGuard(unittest.TestCase):
    """用例会写 ui_prefs.json：整文件备份/还原1092合并写删不掉键）。"""

    def setUp(self):
        p = _prefs_path()
        self._prefs_raw = p.read_bytes() if p.is_file() else None
        self.addCleanup(self._restore_prefs)

    def _restore_prefs(self):
        p = _prefs_path()
        if self._prefs_raw is None:
            if p.is_file():
                p.unlink()
        else:
            p.write_bytes(self._prefs_raw)


# ============================================================ 标注平台
def _main_window(dataset_dir):
    from labelme.app import MainWindow
    from labelme.config import get_default_config

    win = MainWindow(config=get_default_config())
    win.importDirImages(str(dataset_dir), load=True)
    return win


class SmartAnnotateSingleTest(_PrefsGuard):
    """L 键：原实现引用未定义的 labelme.utils / Path，按下必 NameError。"""

    def setUp(self):
        super().setUp()
        from tests.synthetic import make_labelme_dataset

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ds, _ = make_labelme_dataset(Path(self._tmp.name) / "ds",
                                          n_images=2)
        self.weights = Path(self._tmp.name) / "fake.pt"
        self.weights.write_bytes(b"0")

    def _run(self, task_value):
        pred = SimpleNamespace(labels=[0], scores=[0.9],
                               boxes=[[5.0, 5.0, 30.0, 30.0]], masks=None)
        fake = SimpleNamespace(task=SimpleNamespace(value=task_value),
                               classes=["cat"], predict=lambda arr: pred)
        win = _main_window(self.ds)
        self.addCleanup(win.close)
        win._config["ai"] = {"batch_model": str(self.weights)}
        n0 = len(win.canvas.shapes)
        with mock.patch("xccv.engine.models.registry.load_model",
                        return_value=fake):
            win._act_smart.trigger()
            done = _pump_until(lambda: "智能标注" in
                               win.statusBar().currentMessage()
                               and "推理中" not in
                               win.statusBar().currentMessage())
        self.assertTrue(done, "智能标注未结束")
        return win, n0

    def test_detection_model_appends_shapes(self):
        win, n0 = self._run("detection")
        self.assertIn("智能标注完成", win.statusBar().currentMessage())
        self.assertEqual(len(win.canvas.shapes), n0 + 1)

    def test_classification_model_rejected_with_message(self):
        win, n0 = self._run("classification")
        self.assertIn("需要检测/分割模型", win.statusBar().currentMessage())
        self.assertEqual(len(win.canvas.shapes), n0)


class AutoAnnotateDialogTest(_PrefsGuard):
    """批量预标注：auto_cleanup_thread 缺 thread 参数，点「开始」即 TypeError。"""

    def test_start_runs_and_remembers_paths(self):
        from xccv.annotation import auto_annotate
        from xccv.annotation.auto_annotate_window import AutoAnnotateDialog

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = Path(tmp.name) / "m.pt"
        model.write_bytes(b"0")

        def fake_runner(progress_cb=None, cancel=None, **kw):
            progress_cb(1, 1, "a.jpg")
            return {"total": 1, "annotated": 1, "skipped": 0,
                    "cancelled": False, "failed": 0, "images": []}

        dlg = AutoAnnotateDialog()
        self.addCleanup(dlg.close)
        dlg.model_edit.setText(str(model))
        dlg.folder_edit.setText(tmp.name)
        with mock.patch.object(auto_annotate, "annotate_folder", fake_runner):
            dlg.btn_start.click()
            ok = _pump_until(lambda: dlg.status.text().startswith("完成"))
        self.assertTrue(ok, dlg.status.text())
        self.assertEqual(dlg.bar.value(), 100)
        dlg2 = AutoAnnotateDialog()                 # 记住上次的模型/文件夹
        self.addCleanup(dlg2.close)
        self.assertEqual(dlg2.model_edit.text(), str(model))
        self.assertEqual(dlg2.folder_edit.text(), tmp.name)


def _canvas(mode):
    from labelme.widgets.canvas import Canvas

    cv = Canvas()
    cv.resize(200, 100)
    cv.pixmap = QtGui.QPixmap(200, 100)
    cv.pixmap.fill(QtGui.QColor(255, 255, 255))
    cv.createMode = mode
    cv.mode = Canvas.CREATE
    return cv


def _move(cv, x, y):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(x, y),
                           QtCore.Qt.NoButton, QtCore.Qt.NoButton,
                           QtCore.Qt.NoModifier)
    cv.mouseMoveEvent(ev)


class BrushRegionRepaintTest(unittest.TestCase):
    """笔刷框延迟：画笔移动不再同步整幅 repaint()，改局部区域 update。"""

    def test_brush_hover_does_not_full_repaint(self):
        cv = _canvas("brush")
        calls = []
        cv.repaint = lambda *a: calls.append(1)
        _move(cv, 40, 30)
        _move(cv, 60, 50)
        self.assertEqual(calls, [])
        want = cv.transformPos(QtCore.QPointF(60, 50))
        got = cv._brush_last_cursor
        self.assertAlmostEqual(got.x(), want.x(), places=3)
        self.assertAlmostEqual(got.y(), want.y(), places=3)

    def test_other_modes_keep_repaint(self):
        cv = _canvas("rectangle")
        calls = []
        cv.repaint = lambda *a: calls.append(1)
        _move(cv, 40, 30)
        self.assertEqual(calls, [1])

    def test_width_change_resets_cursor_cache(self):
        cv = _canvas("brush")
        _move(cv, 40, 30)
        self.assertIsNotNone(cv._brush_last_cursor)
        cv.setBrushWidth(30)
        self.assertIsNone(cv._brush_last_cursor)


class DoubleClick250Test(unittest.TestCase):
    """全平台双击判定 250ms。"""

    def test_ai_pending_click_window_is_250ms(self):
        from labelme.shape import Shape

        cv = _canvas("ai_polygon")
        cv.current = Shape(shape_type="points")
        cv._ai_prompts = [(QtCore.QPointF(5, 5), 1)]
        cv._ai_click(QtCore.QPointF(50, 50), 1, QtCore.QPointF(50, 50),
                     QtCore.Qt.NoModifier)
        try:
            self.assertTrue(cv._ai_click_timer.isActive())
            self.assertEqual(cv._ai_click_timer.interval(), 250)
        finally:
            cv._ai_click_timer.stop()

    def test_qt_global_and_web_tools(self):
        import inspect

        from xccv import platform_entry

        self.assertIn("setDoubleClickInterval(250)",
                      inspect.getsource(platform_entry.run_platform))
        js = (ROOT / "xccv/defectgen/web/js/interact.js").read_text(
            encoding="utf-8")
        self.assertIn("now - lastPolyClick.t < 250", js)
        lg = (ROOT / "xccv/modeltools/web/litegraph.js").read_text(
            encoding="utf-8")
        self.assertIn("now - this.last_mouseclick < 250", lg)


class KeepScaleToggleTest(_PrefsGuard):
    """实测补充：切图保留上一张缩放且无法关闭——按文件缩放记忆须服从开关。"""

    def test_toggle_controls_per_file_zoom_memory(self):
        from tests.synthetic import make_labelme_dataset
        from xccv.ui_widgets import load_prefs

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ds, _ = make_labelme_dataset(Path(tmp.name) / "ds", n_images=2)
        win = _main_window(ds)
        self.addCleanup(win.close)
        a, b = str(ds / "img00.png"), str(ds / "img01.png")

        win.enableKeepPrevScale(False)
        win.loadFile(a)
        win.setZoom(250)
        win.loadFile(b)
        win.loadFile(a)
        self.assertNotAlmostEqual(win.canvas.scale, 2.5, places=2,
                                  msg="关闭时回到 A 应重新适应窗口")
        self.assertIs(load_prefs().get("keep_prev_scale"), False)

        win.enableKeepPrevScale(True)
        win.loadFile(a)
        win.setZoom(250)
        win.loadFile(b)
        win.loadFile(a)
        self.assertAlmostEqual(win.canvas.scale, 2.5, places=2)
        self.assertIs(load_prefs().get("keep_prev_scale"), True)


# ============================================================ 训练平台
class AnchorSmallChainTest(unittest.TestCase):
    """小目标锚框：Project → TrainThread → TrainConfig → 网络 → ckpt 往返。"""

    def test_trainconfig_default_off(self):
        from xccv.engine.models.base import TrainConfig

        cfg = TrainConfig(task="detection", dataset_dir=".", output_dir=".",
                          model_name="x")
        self.assertFalse(cfg.anchor_small)

    def test_trainer_thread_passes_flag(self):
        from xccv.training.dataset_stats import DatasetStats
        from xccv.training.project import Project
        from xccv.training.trainer_thread import TrainThread

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        seen = {}

        class _FakeModel:
            def train(self, cfg, reporter):
                seen["cfg"] = cfg
                return {"best_val_map50": 0.0}

        p = Project(name="t", dataset_dir=tmp.name, model_dir=tmp.name,
                    task="segmentation", model_name="maskrcnn18",
                    anchor_small=True)
        th = TrainThread(p, Path(tmp.name) / "run")
        stats = DatasetStats(classes=["a"], n_images=1)
        with mock.patch("xccv.training.trainer_thread.collect_stats",
                        return_value=stats), \
                mock.patch("xccv.engine.models.create_model",
                           return_value=_FakeModel()):
            th.run()
        self.assertTrue(seen["cfg"].anchor_small)

    def test_anchor_pyramid_and_ckpt_roundtrip(self):
        from xccv.engine.models.rcnn import RCNNModel, _anchor_base

        self.assertEqual(_anchor_base(640), 8)
        self.assertEqual(_anchor_base(1280), 16)
        m = RCNNModel("maskrcnn18", "segmentation", 2, device="cpu",
                      use_pretrained=False)
        self.assertEqual(m.net.rpn.anchor_generator.sizes[0], (32,))
        net = m._build("maskrcnn18", 2, 320, False, anchor_base=16.0)
        want = tuple((16.0 * 2 ** i,) for i in range(5))
        self.assertEqual(tuple(net.rpn.anchor_generator.sizes), want)
        ckpt = {"format": "xccv-rcnn", "arch": "maskrcnn18",
                "task": "segmentation", "classes": ["a", "b"],
                "min_size": 320, "state_dict": net.state_dict(),
                "industrial_opt": False, "mask_head_light": False,
                "slim_head": False, "anchor_base": 16.0}
        m2 = RCNNModel("fasterrcnn", "detection", 1, device="cpu",
                       use_pretrained=False)
        m2.load_weights(ckpt)
        self.assertEqual(tuple(m2.net.rpn.anchor_generator.sizes), want)


class ProjectOrderTest(unittest.TestCase):
    """项目卡片按最近更新时间倒序（project.json 与各次训练 metrics.json）。"""

    def test_order_by_latest_touch(self):
        from tests.synthetic import make_labelme_dataset
        from xccv.training import project as project_mod

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ds, _ = make_labelme_dataset(Path(tmp.name) / "ds", n_images=1)
        with mock.patch.object(project_mod, "PROJECTS_DIR",
                               Path(tmp.name) / "projects"):
            a = project_mod.create_project("甲", str(ds), "", "detection")
            b = project_mod.create_project("乙", str(ds), "", "detection")
            os.utime(a.project_dir / "project.json", (1000, 1000))
            os.utime(b.project_dir / "project.json", (2000, 2000))
            self.assertEqual(project_mod.list_projects(), ["乙", "甲"])
            run = a.project_dir / "models" / "20261107_100000_maskrcnn18"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            os.utime(run / "metrics.json", (3000, 3000))
            self.assertEqual(project_mod.list_projects(), ["甲", "乙"])




class AnchorCheckboxTest(_TrainEnv):
    """训练窗口「小目标锚框自适应」复选框：加载/保存/持久化。"""

    def test_checkbox_persists(self):
        from xccv.training.project import load_project

        pw = self._window()
        self.assertFalse(pw.par_anchor.isChecked())
        pw.par_anchor.setChecked(True)
        pw._save_params()
        self.assertTrue(load_project("ui").anchor_small)
        pw2 = self._window()
        self.assertTrue(pw2.par_anchor.isChecked())


# ============================================================ 模型工具
class DefectgenPortTest(unittest.TestCase):
    """固定优先端口（IndexedDB 按 origin 分区，端口漂移=资产清空）；同进程
    多实例不得撞端口（探测释放后服务线程才监听的竞态）。"""

    def test_claims_distinct_and_release(self):
        code = ("from xccv.defectgen.defectgen_window import _pick_port, "
                "_release_port; a=_pick_port(); b=_pick_port(); "
                "_release_port(a); c=_pick_port(); print(a, b, c)")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                           env=env, capture_output=True, text=True,
                           timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        a, b, c = (int(x) for x in r.stdout.split()[-3:])
        self.assertTrue(39771 <= a < 39771 + 48)
        self.assertNotEqual(a, b)
        self.assertEqual(c, a)


class AccelUiTest(unittest.TestCase):
    """模型加速：源权重自动建议输入尺寸 + 导出进度条（忙碌/百分比）。"""

    def test_suggest_imgsz_variants(self):
        from xccv.modeltools.modeltools_window import _suggest_imgsz

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        torch.save({"min_size": 512}, d / "a.pt")
        self.assertEqual(_suggest_imgsz(d / "a.pt"), 512)
        torch.save({"train_input": 0, "image_size": 640}, d / "b.pt")
        self.assertEqual(_suggest_imgsz(d / "b.pt"), 640)
        torch.save({"train_args": {"imgsz": 736}}, d / "c.pt")
        self.assertEqual(_suggest_imgsz(d / "c.pt"), 736)
        (d / "bad.pt").write_bytes(b"junk")
        self.assertIsNone(_suggest_imgsz(d / "bad.pt"))

    def test_progress_bar_states(self):
        from xccv.modeltools.modeltools_window import AccelerateToolWindow

        w = AccelerateToolWindow()
        self.addCleanup(w.close)
        w._on_export_progress(-1, "构建中")
        self.assertEqual((w.accel_bar.minimum(), w.accel_bar.maximum()),
                         (0, 0))
        w._on_export_progress(0.5, "一半")
        self.assertEqual(w.accel_bar.maximum(), 100)
        self.assertEqual(w.accel_bar.value(), 50)
        self.assertEqual(w.accel_result.text(), "一半")


# ============================================================ YOLO（anchor-free）
def _case(cx_off_px, stride=8.0):
    """微小目标 GT 7.5×5px，预测中心偏移 cx_off_px（特征尺度坐标）。"""
    tw, th = 7.5 / stride, 5.0 / stride
    tb = torch.tensor([[100.0 - tw / 2, 50.0 - th / 2,
                        100.0 + tw / 2, 50.0 + th / 2]])
    pb = tb.clone()
    pb[0, 0] += cx_off_px / stride
    pb[0, 2] += cx_off_px / stride
    pb.requires_grad_(True)
    pd = torch.full((1, 64), 8.0, requires_grad=True)
    return (pd, pb, torch.tensor([[100.0, 50.0]]), tb,
            torch.tensor([[[0.9]]]), torch.tensor(0.9),
            torch.tensor([True]), torch.tensor([960.0, 960.0]),
            torch.full((1, 1), stride))


class NWDLossTest(unittest.TestCase):
    """旧 Focaler-WIoU 以 GT 尺寸归一再 exp：微小目标 loss 1.5e6 且不重叠时
    框梯度为 0。新损失有界、恒有梯度。"""

    def setUp(self):
        from xccv.engine.models.modules import NWDCIoUBboxLoss

        self.loss = NWDCIoUBboxLoss(reg_max=16)

    def test_bounded_with_gradient_when_far(self):
        args = _case(60.0)
        li, ld = self.loss(*args)
        (li + ld).backward()
        self.assertLessEqual(float(li), 3.0)
        g = float(args[1].grad.abs().sum())
        self.assertTrue(0.0 < g < 1e3, g)

    def test_perfect_prediction_near_zero(self):
        li, _ = self.loss(*_case(0.0))
        self.assertLess(float(li), 0.05)

    def test_batched_2d_fg_mask(self):
        anchors = torch.tensor([[50.0, 50.0], [150.0, 50.0],
                                [50.0, 150.0], [150.0, 150.0]])
        stride = torch.tensor([[8.0], [8.0], [16.0], [16.0]])
        tb = torch.zeros(2, 4, 4)
        tb[0, 0] = torch.tensor([49.0, 49.4, 49.9, 50.0])
        tb[1, 1] = torch.tensor([149.0, 149.4, 149.9, 150.0])
        pb = tb.clone()
        pb[0, 0, [0, 2]] += 60 / 8
        pb[1, 1, [0, 2]] += 60 / 16
        pb.requires_grad_(True)
        ts = torch.zeros(2, 4, 1)
        ts[0, 0, 0] = ts[1, 1, 0] = 0.9
        fg = torch.tensor([[True, False, False, False],
                           [False, True, False, False]])
        li, ld = self.loss(torch.full((2, 4, 64), 8.0), pb, anchors, tb, ts,
                           torch.tensor(1.8), fg,
                           torch.tensor([960.0, 960.0]), stride)
        li.backward()
        self.assertTrue(torch.isfinite(li))
        self.assertLessEqual(float(li), 3.0)
        self.assertTrue(torch.isfinite(pb.grad).all())

    def test_yolo_industrial_patch_uses_nwd_only(self):
        from xccv.engine.models.modules import NWDCIoUBboxLoss
        from xccv.engine.models.yolo import YoloModel

        class _Net(torch.nn.Linear):
            def init_criterion(self):
                return SimpleNamespace(reg_max=16, bbox_loss="native",
                                       bce="native-bce")

        net = _Net(1, 1)
        me = SimpleNamespace(_loss_patch="off")
        YoloModel._apply_industrial_loss(me, SimpleNamespace(model=net))
        crit = net.init_criterion()
        self.assertIsInstance(crit.bbox_loss, NWDCIoUBboxLoss)
        self.assertEqual(crit.bce, "native-bce")      # cls 走原生
        self.assertTrue(me._loss_patch.startswith("applied(NWD"))


def _val_data():
    shp = SimpleNamespace(label="a",
                          points=[[10, 10], [30, 10], [30, 30], [10, 30]])
    smp = SimpleNamespace(width=100, height=100, image_path="a.jpg",
                          shapes=[shp])
    return SimpleNamespace(val_samples=[smp], classes=["a"])


class YoloLooseMetricTest(unittest.TestCase):
    """宽松口径：YOLO 路径此前不产出 map_loose（训练表恒 0）；逐轮评估复用
    验证期已 NMS 的检出（on_val_start 挂不上 trainer 回调）。"""

    def test_validator_wrap_collects_scaled_dets(self):
        from xccv.engine.models.yolo import YoloModel

        class _V:
            calls = 0

            def _prepare_pred(self, pred):
                return pred

            def _prepare_batch(self, si, batch):
                return {"im_file": batch["im_file"][si]}

            def scale_preds(self, predn, pbatch):
                return predn

            def update_metrics(self, preds, batch):
                _V.calls += 1

        v = _V()
        me = SimpleNamespace(_val_index={"a.jpg": 0, "ok.jpg": 1},
                             _epoch_dets=[], _det_seen=0)
        YoloModel._on_val_start(me, SimpleNamespace(validator=v))
        YoloModel._on_val_start(me, SimpleNamespace(validator=v))  # 不重复包
        preds = [{"bboxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                  "conf": torch.tensor([0.9]), "cls": torch.tensor([0.0])},
                 # OK 负样本：无 GT 且无检出（此前整批检出被丢）
                 {"bboxes": torch.zeros((0, 4)), "conf": torch.zeros(0),
                  "cls": torch.zeros(0)}]
        batch = {"im_file": ["a.jpg", "ok.jpg"],
                 "img": torch.zeros((2, 3, 100, 100)),
                 "ori_shape": [(100, 100), (100, 100)],
                 "ratio_pad": [((1.0, 1.0), (0.0, 0.0)),
                               ((1.0, 1.0), (0.0, 0.0))]}
        v.update_metrics(preds, batch)
        self.assertEqual(_V.calls, 1)
        self.assertEqual(len(me._epoch_dets), 1)
        det = me._epoch_dets[0]
        self.assertEqual((det["image_id"], det["category_id"]), (0, 1))
        self.assertEqual(det["bbox"], [10.0, 10.0, 20.0, 20.0])
        YoloModel._on_epoch_dets_reset(me, None)
        self.assertEqual(me._epoch_dets, [])

    def test_epoch_loose_values(self):
        from xccv.engine.models.yolo import YoloModel

        hit = {"image_id": 0, "category_id": 1,
               "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9}
        me = SimpleNamespace(_val_data=_val_data(), _epoch_dets=[hit])
        out = YoloModel._epoch_loose(me)
        self.assertAlmostEqual(out["map_loose"], 1.0, places=3)
        self.assertAlmostEqual(out["mar_loose"], 1.0, places=3)
        me._epoch_dets = []
        self.assertEqual(YoloModel._epoch_loose(me)["map_loose"], 0.0)

    def test_final_loose_with_predict_fn(self):
        from xccv.engine.models.yolo import YoloModel

        box = SimpleNamespace(xyxy=torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
                              cls=torch.tensor(0.0), conf=torch.tensor(0.9))
        out = YoloModel._loose_metrics(
            SimpleNamespace(), _val_data(),
            predict_fn=lambda p: SimpleNamespace(boxes=[box]))
        self.assertAlmostEqual(out["map_loose"], 1.0, places=3)

# == 来源模块：test_v06_b4_fixes ==
# -*- coding: utf-8 -*-
"""回归：训练几何修复（Y2）、OK 负样本训练、P2 检测头、旧字样清零。

- YoloGeometryTest：训练取样与验证/推理同一几何（长边 imgsz 保持长宽比），
  框像素长宽比不再被非等比拉伸（Y1geo 对照实验：修复宽松 mAP +67%）；
- NegativeSampleTest：负样本文件夹（OK/良品/负样本/背景）内未标注图进入
  训练与验证，标注图划分不变，超量自动抽样，训练产物含误检统计键；
- P2HeadTest / P2CheckboxTest：P2 小目标头作为训练参数（仅 YOLO 家族），
  模板构建、检测头按步长对齐迁移、训练线程接线、复选框持久化与预检提示；
- LegacyWordingTest：平台代码中旧工具字样（注释/字符串）零残留。
"""

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(ROOT))

from xccv.engine.dataset.base import TaskType
from xccv.engine.dataset.reader import (
    NEGATIVE_DIR_NAMES, is_negative_dir_name, scan_negative_images)
from xccv.engine.dataset.prepare import prepare
from xccv.engine.models.base import negative_fp_stats
from xccv.engine.models.registry import MODEL_SPECS, create_model
from xccv.training.preflight import param_findings

import numpy as np

from tests.synthetic import make_labelme_dataset


def _write_min_jpg(path: Path, w: int = 64, h: int = 48):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


class YoloGeometryTest(unittest.TestCase):
    """训练分支与验证分支同一几何（长边缩放，保持长宽比）。"""

    @classmethod
    def setUpClass(cls):
        from ultralytics.cfg import get_cfg

        from xccv.engine.models.yolo_data import InMemoryYoloDataset

        cls._ds_cls = InMemoryYoloDataset
        hyp = get_cfg()
        for k in ("mosaic", "mixup", "copy_paste", "cutmix", "degrees",
                  "translate", "scale", "shear", "perspective", "flipud",
                  "fliplr", "bgr", "hsv_h", "hsv_s", "hsv_v", "erasing"):
            if hasattr(hyp, k):
                setattr(hyp, k, 0.0)
        cls._hyp = hyp
        tmp = cls._tmp = __import__("tempfile").TemporaryDirectory()
        root, classes = make_labelme_dataset(
            Path(tmp.name) / "ds", n_images=2, size=(128, 64),
            shape_type="rectangle", classes=("cat",))
        samples, cls.classes, _ = __import__(
            "xccv.engine.dataset.reader", fromlist=["x"]).scan_labelme_dataset(root)
        cls.samples = samples

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _ds(self, rect: bool):
        return self._ds_cls(
            self.samples, self.classes, "detection", None,
            img_path="xccv://in-memory", imgsz=64, batch_size=2,
            augment=not rect, hyp=self._hyp, rect=rect, cache=None,
            single_cls=False, stride=32, pad=0.0 if not rect else 0.5,
            prefix="", data={"channels": 3}, fraction=1.0)

    def test_train_keeps_aspect_ratio(self):
        """训练（rect=False）：非方形图长边缩放到 imgsz、保持长宽比
        （此后 ultralytics 训练变换只做 letterbox 补边，不再变形）。"""
        ds = self._ds(rect=False)
        lab = ds.get_image_and_label(0)
        h, w = lab["img"].shape[:2]
        self.assertEqual(max(h, w), 64)
        self.assertAlmostEqual(w / h, 2.0, delta=0.1,
                               msg="128x64 图应缩成 64x32（不再拉伸成 64x64）")
        self.assertEqual(tuple(lab["resized_shape"]), (h, w))

    def test_boxes_pixel_aspect_preserved(self):
        """整条训练管线后框的像素长宽比与原图一致（旧实现拉伸 ×4）。"""
        ds = self._ds(rect=False)
        item = ds[0]
        H, W = item["img"].shape[-2:]
        got = np.asarray(item["bboxes"], dtype=np.float32).reshape(-1, 4)
        base = ds.labels[0]["bboxes"]
        ar_in = (got[:, 2] * W) / (got[:, 3] * H)
        ar_src = (base[:, 2] * 128.0) / (base[:, 3] * 64.0)
        self.assertTrue(np.allclose(ar_in, ar_src, rtol=0.1),
                        f"{ar_in} vs {ar_src}")

    def test_val_rect_same_geometry(self):
        """验证（rect=True）与训练取样几何一致。"""
        lab_v = self._ds(rect=True).get_image_and_label(0)
        lab_t = self._ds(rect=False).get_image_and_label(0)
        self.assertEqual(lab_v["img"].shape[:2], lab_t["img"].shape[:2])

    def test_train_letterbox_pad_114(self):
        """与 ultralytics 官方一致：训练图经 LetterBox 补 114 灰边成 imgsz
        方形（不拉伸）。128x64 → 内容 64x32，上下各补 16 行。"""
        item = self._ds(rect=False)[0]
        img = item["img"]          # (3, 64, 64) uint8 tensor
        self.assertEqual(tuple(img.shape[-2:]), (64, 64))
        top = img[:, :8, :].float()
        self.assertTrue(bool((top == 114).all()), "补边应为 114 灰")

    def test_resize_op_not_applied(self):
        """YOLO 训练忽略 resize 算子的非等比拉伸（只取长边作网络输入）。"""
        ds = self._ds_cls(
            self.samples, self.classes, "detection",
            {"resize": {"width": 32, "height": 320}},
            img_path="xccv://in-memory", imgsz=64, batch_size=2,
            augment=True, hyp=self._hyp, rect=False, cache=None,
            single_cls=False, stride=32, pad=0.0, prefix="",
            data={"channels": 3}, fraction=1.0)
        h, w = ds.get_image_and_label(0)["img"].shape[:2]
        self.assertAlmostEqual(w / h, 2.0, delta=0.1,
                               msg="不应被 resize 算子拉成 32x320")


class NegativeSampleTest(unittest.TestCase):
    """OK 图负样本：扫描、划分、倍增与误检统计。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = __import__("tempfile").TemporaryDirectory()
        root = Path(cls.tmp.name) / "ds"
        root.mkdir(parents=True)
        _, _ = make_labelme_dataset(
            root / "缺陷", n_images=6, size=(64, 48), classes=("cat", "dog"))
        for i in range(5):  # OK 负样本：5 张未标注图
            _write_min_jpg(root / "OK" / f"ok_{i}.jpg")
        _write_min_jpg(root / "OK" / "with_json.jpg")   # OK 目录但有标注 → 非负样本
        _write_min_jpg(root / "缺陷" / "unlabeled.jpg")  # 缺陷目录未标注 → 不收
        (root / "OK" / "with_json.json").write_text(
            '{"imagePath": "with_json.jpg", "imageWidth": 64, "imageHeight": 48,'
            ' "shapes": [{"label": "cat", "shape_type": "rectangle",'
            ' "points": [[1, 1], [10, 10]]}]}', encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dir_names_and_scan(self):
        self.assertTrue(is_negative_dir_name("OK") and is_negative_dir_name("良品")
                        and is_negative_dir_name("负样本"))
        self.assertFalse(is_negative_dir_name("黑点"))
        negs = scan_negative_images(Path(self.tmp.name) / "ds")
        self.assertEqual(len(negs), 5, "只收 OK 目录内无 json 的 5 张")
        self.assertTrue(all(not s.shapes for s in negs))

    def test_prepare_splits_negatives_and_keeps_positives(self):
        root = Path(self.tmp.name) / "ds"
        off = prepare(TaskType.DETECTION, "yolo", root, val_ratio=0.2,
                      seed=42, train_ratio=0.8)
        on = prepare(TaskType.DETECTION, "yolo", root, val_ratio=0.2,
                     seed=42, train_ratio=0.8, use_negatives=True)
        self.assertEqual(off.neg_train_count, 0)
        self.assertGreater(on.neg_train_count, 0)
        self.assertGreater(on.neg_val_count, 0)
        pos_off = [s for s in off.train_samples if s.shapes]
        pos_on = [s for s in on.train_samples if s.shapes]
        self.assertEqual([str(s.image_path) for s in pos_off],
                         [str(s.image_path) for s in pos_on],
                         "标注图划分必须与关闭负样本时完全一致")
        self.assertTrue(all(not s.shapes for s in on.train_samples
                            if s not in pos_on), "负样本应为空标注")

    def test_negative_cap(self):
        """负样本多于标注图 ×1.0 时训练侧抽样（验证侧保留）。"""
        root = Path(self.tmp.name) / "ds"
        data = prepare(TaskType.DETECTION, "yolo", root, val_ratio=0.0,
                       seed=42, train_ratio=1.0, use_negatives=True)
        n_pos_train = len(data.train_samples)
        self.assertLessEqual(data.neg_train_count, n_pos_train)
        self.assertGreater(data.neg_train_count, 0)

    def test_negative_fp_stats(self):
        """误检统计：分数阈值过滤 + 空样本集返回空。"""
        class _P:
            def __init__(self, scores):
                self.scores = scores

        negs = [type("S", (), {"image_path": Path("x.jpg"), "shapes": []})()]
        calls = []

        def fake_predict(img):
            calls.append(1)
            return _P([0.9, 0.1])

        def fake_read(path):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        from unittest import mock

        with mock.patch("xccv.engine.io_utils.imread_unicode", fake_read):
            out = negative_fp_stats(fake_predict, negs, conf=0.25)
        self.assertEqual(out["neg_fp_per_img"], 1.0)  # 0.1 低于阈值不计
        self.assertEqual(out["neg_fp_img_rate"], 1.0)
        self.assertEqual(negative_fp_stats(fake_predict, [], conf=0.25), {})


class P2HeadTest(unittest.TestCase):
    """P2 小目标头＝训练参数（与 slim 头/小目标锚框同一机制，仅 YOLO 家族）：
    enable_p2 切换/模板/预训练迁移/训练线程接线/预检提示。"""

    def test_registry_has_no_p2_names(self):
        """P2 不再是独立模型名：下拉只列基础模型，由复选框叠加。"""
        self.assertFalse([n for n in MODEL_SPECS if n.endswith("-p2")])

    def test_enable_p2_switches_yolo_variants(self):
        for name, task, base in (("yolo11n", TaskType.DETECTION, "yolo11n"),
                                 ("yolo11s", TaskType.DETECTION, "yolo11s"),
                                 ("yolo11n-seg", TaskType.SEGMENTATION, "yolo11n-seg"),
                                 ("yolo11s-seg", TaskType.SEGMENTATION, "yolo11s-seg")):
            m = create_model(name, task, 2, device="cpu", use_pretrained=True)
            self.assertFalse(m.p2)
            m.enable_p2()
            m.enable_p2()   # 幂等
            self.assertTrue(m.p2)
            self.assertEqual(m.p2_source, base)
            self.assertTrue(m.init_weight.endswith(f"{base}-p2.yaml"), m.init_weight)
            self.assertTrue(os.path.isfile(re.sub(
                r"(\d+)([nslmx])(.*)$", r"\1\3", m.init_weight)),
                f"统一模板必须存在: {m.init_weight}")

    def test_trainer_thread_wiring(self):
        """训练线程：项目勾选 P2 → YOLO 模型启用、R-CNN 忽略；快照只记实际生效。"""
        from unittest import mock

        from xccv.training import trainer_thread as T
        from xccv.training.project import Project

        captured = {}

        def fake_train(self, cfg, reporter=None):
            captured["cfg"], captured["model"] = cfg, self
            return {}

        for name, task, expect in (("yolo11n-seg", "segmentation", True),
                                   ("maskrcnn18", "segmentation", False)):
            p = Project(name="p2", dataset_dir=str(ROOT), model_dir=str(ROOT),
                        task=task, model_name=name, p2_head=True)
            th = T.TrainThread(p, ROOT / "_tmp" / "p2_wiring")
            stats = type("S", (), {"n_images": 1, "classes": ["a"]})()
            with mock.patch.object(T, "collect_stats", return_value=stats), \
                    mock.patch("xccv.engine.models.base.BaseModel.train", fake_train):
                th.run()
            self.assertEqual(captured["cfg"].p2_head, expect, name)
            self.assertEqual(bool(getattr(captured["model"], "p2", False)), expect)

    def test_templates_have_p2_level(self):
        import yaml

        base = ROOT / "xccv" / "engine" / "models" / "configs"
        for f in ("yolo11-p2.yaml", "yolo11-seg-p2.yaml"):
            data = yaml.safe_load((base / f).read_text(encoding="utf-8"))
            head = data["head"]
            detect = head[-1]
            self.assertEqual(len(detect[0]), 4, f"{f} 检测层应为 4 个（含 P2）")
            self.assertEqual(detect[2], "Segment" if "seg" in f else "Detect")
            # P2 分支必须引用骨干第 2 层（stride 4）
            cat_inputs = [h for h in head if h[2] == "Concat" and 2 in h[0]]
            self.assertTrue(cat_inputs, f"{f} 缺少 P2（backbone 层2）Concat")
            if "seg" in f:
                # proto 必须取 stride-8 特征（层16）：Proto 仅上采样 ×2，
                # 掩码须与 GT mask_ratio=4 同步（取 P2 会 matmul 形状崩）
                self.assertEqual(head[-1][0][0], 16, f"{f} proto 输入应为层16")

    def test_shift_head_levels(self):
        """检测头按步长对齐迁移：目标 stride 8/16/32 分支逐张量等于源
        P3/P4/P5 分支；新增 stride 4 分支以源最细分支（stride 8）初始化
        （随机初始化收敛极慢）——检测模板输入 [P2,P3,P4,P5] 与分割模板
        [P3,P2,P4,P5]（proto 取 P3）都必须对齐。"""
        torch = __import__("torch")
        from ultralytics import YOLO

        from xccv.engine.offline import WEIGHTS_DIR
        from xccv.engine.models.yolo import _shift_head_levels

        cfg_dir = ROOT / "xccv" / "engine" / "models" / "configs"
        for base_name in ("yolo11n", "yolo11n-seg"):
            base = Path(WEIGHTS_DIR) / "ultralytics" / f"{base_name}.pt"
            if not base.is_file():
                self.skipTest("随包权重缺失")
            net = YOLO(str(cfg_dir / f"{base_name}-p2.yaml")).model
            src = YOLO(str(base)).model
            head, shead = net.model[-1], src.model[-1]
            s_dst = [int(s) for s in head.stride]
            s_src = [int(s) for s in shead.stride]
            moved = _shift_head_levels(net, src)
            self.assertGreater(moved, 100, base_name)
            pairs = [(s_dst.index(s), i, s) for i, s in enumerate(s_src)]
            pairs.append((s_dst.index(4), s_src.index(8), 4))  # 新增 P2 ← 源 P3
            for j, i, s in pairs:
                for branch in ("cv2", "cv3"):
                    a = getattr(head, branch)[j].state_dict()
                    b = getattr(shead, branch)[i].state_dict()
                    same = [k for k in b if k in a and a[k].shape == b[k].shape]
                    self.assertTrue(same, (base_name, branch, s))
                    self.assertTrue(all(torch.equal(a[k], b[k].float()) for k in same),
                                    f"{base_name} {branch} 目标 stride{s} 分支来源不对")

    def test_preflight_p2_hint(self):
        from xccv.training.project import Project

        def codes(model_name, p2):
            p = Project(name="x", dataset_dir="d", model_dir="m",
                        task="detection", model_name=model_name, p2_head=p2)
            return {f.code for f in param_findings(p, None, cuda_available=False)}

        self.assertIn("p2_vram", codes("yolo11n", True))
        self.assertIn("p2_not_yolo", codes("fasterrcnn", True))
        self.assertFalse({"p2_vram", "p2_not_yolo"} & codes("yolo11n", False))




class P2CheckboxTest(_TrainEnv):
    """训练窗口「P2 小目标检测头（仅 YOLO 家族生效）」复选框：加载/保存/持久化。"""

    def test_checkbox_persists(self):
        from xccv.training.project import load_project

        pw = self._window()
        self.assertIn("YOLO", pw.par_p2.text())
        self.assertFalse(pw.par_p2.isChecked())
        pw.par_p2.setChecked(True)
        pw._save_params()
        self.assertTrue(load_project("ui").p2_head)
        pw2 = self._window()
        self.assertTrue(pw2.par_p2.isChecked())


class LegacyWordingTest(unittest.TestCase):
    """平台代码（xccv/ 与 tests/）中旧工具字样零残留（含注释）。"""

    def test_no_legacy_wording(self):
        pat = re.compile("DL" + "CV", re.IGNORECASE)   # 拼接书写：本文件自身不命中
        hits = []
        for start in ("xccv", "tests"):
            for dp, dn, fn in os.walk(ROOT / start):
                dn[:] = [d for d in dn if d not in ("__pycache__",)]
                for f in fn:
                    if not f.endswith((".py", ".html", ".js", ".qss")):
                        continue
                    text = (Path(dp) / f).read_text(encoding="utf-8",
                                                    errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pat.search(line):
                            hits.append(f"{Path(dp, f)}:{i}")
        self.assertEqual(hits, [], f"平台代码仍含旧字样: {hits[:8]}")

# == 来源模块：test_batch11_ui ==
"""UI 测试：齿轮开发者选项、体检/预检对话框、预检接线、launcher 透传、字号下限。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.training import dev_options as dev_mod
from xccv.training import project as project_mod
from xccv.training.data_health import HealthIssue, LEVEL_INFO, LEVEL_WARN
from xccv.training.dev_options import DevOptions
from xccv.training.preflight import Finding
from xccv.training.project import create_project, load_project

_APP__batch11_ui = QApplication.instance() or QApplication(sys.argv)


class _TempEnv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        p = mock.patch.object(dev_mod, "options_path",
                              return_value=Path(self._tmp.name) / "dev_options.json")
        p.start()
        self.addCleanup(p.stop)
        det = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)[0]
        create_project("ui", str(det), "", "detection")


class CollapsibleSectionTest(_TempEnv):
    def test_toggle(self):
        from xccv.training.dialogs import CollapsibleSection

        sec = CollapsibleSection("测试分区", expanded=True)
        self.assertFalse(sec.content.isHidden())  # 初始展开
        sec.set_expanded(False)                   # 折叠
        self.assertTrue(sec.content.isHidden())
        sec.toggle()                              # 再展开
        self.assertFalse(sec.content.isHidden())
        self.assertIn("▾", sec.btn.text())


class HealthReportDialogTest(_TempEnv):
    def _rows(self, with_finding=True):
        rows = [HealthIssue(LEVEL_WARN, "imbalance", "类别不均衡", "3:1",
                            "建议开启工业优化")]
        if with_finding:
            rows.append(Finding(LEVEL_INFO, "val_zero", "验证集比例为 0"))
            rows.append(Finding(LEVEL_WARN, "patience", "早停不会触发",
                                field="patience", value=5, apply_text="设为 5 轮"))
        return rows

    def test_report_mode_builds(self):
        from xccv.training.dialogs import HealthReportDialog

        dlg = HealthReportDialog(None, "数据体检报告", self._rows(False),
                                 mode="report", summary="测试")
        self.assertEqual(dlg.mode, "report")
        dlg.close()

    def test_preflight_apply_records(self):
        from xccv.training.dialogs import HealthReportDialog

        applied = []
        dlg = HealthReportDialog(None, "训练前预检", self._rows(True),
                                 mode="preflight",
                                 on_apply=lambda k, v: applied.append((k, v)))
        self.assertTrue(dlg._btn_apply_all.isVisibleTo(dlg))
        dlg._apply_all()
        self.assertEqual(applied, [("patience", 5)])
        self.assertEqual(dlg.applied, [("patience", 5)])
        self.assertIn("早停耐心 → 5", dlg.applied_label.text())
        dlg.close()

    def test_no_warn_info_only_preflight(self):
        """仅提示级：对话框无「应用全部」也仍可开始（弹窗逻辑在窗口层判断）。"""
        from xccv.training.dialogs import HealthReportDialog

        rows = [HealthIssue(LEVEL_INFO, "val_zero", "验证集比例为 0")]
        dlg = HealthReportDialog(None, "训练前预检", rows, mode="preflight")
        self.assertFalse(dlg._btn_apply_all.isVisibleTo(dlg))
        dlg.close()


class DevOptionsDialogTest(_TempEnv):
    def test_save_roundtrip(self):
        from xccv.training.dialogs import DevOptionsDialog

        dlg = DevOptionsDialog()
        dlg.sp_imbalance.setValue(4.0)  # 对话框单位是百分比
        dlg.sp_cls.setValue(15)
        dlg.ck_preflight.setChecked(False)
        dlg.ck_info_popup.setChecked(False)
        dlg._save()   # 不走模态 exec_
        opt = dev_mod.load_options()
        self.assertEqual(opt.imbalance_share, 0.04)
        self.assertEqual(opt.min_cls_samples, 15)
        self.assertFalse(opt.preflight_enabled)
        self.assertFalse(opt.preflight_info_popup)

    def test_reset_defaults(self):
        from xccv.training.dialogs import DevOptionsDialog

        dlg = DevOptionsDialog()
        dlg.sp_imbalance.setValue(20.0)
        dlg._reset_defaults()
        self.assertEqual(dlg.sp_imbalance.value(), 5.0)


class PreflightWiringTest(_TempEnv):
    """预检接线：仅提示级不弹窗直接开始；警告级弹窗（stub）确认后开始。"""

    def _window(self):
        from xccv.training.training_window import ProjectWindow

        pw = ProjectWindow(load_project("ui"))
        self.addCleanup(pw.close)
        return pw

    def test_apply_preflight_value_mapping(self):
        pw = self._window()
        pw._apply_preflight_value("patience", 7)
        pw._apply_preflight_value("learning_rate", 0.01)
        pw._apply_preflight_value("batch_size", 8)
        self.assertEqual(pw.par_patience.value(), 7)
        self.assertEqual(pw.par_lr.value(), 0.01)
        self.assertEqual(pw.par_batch.value(), 8)

    def test_apply_resize_value(self):
        pw = self._window()
        pw.augment_controls["resize"][0].setChecked(False)
        pw._apply_preflight_value("resize", (512, 384))
        check, spins = pw.augment_controls["resize"]
        self.assertTrue(check.isChecked())
        self.assertEqual(pw._current_resize(), (512, 384))

    def test_info_only_preflight_starts_directly(self):
        """开发者选项关闭"提示级弹窗"时：仅提示不弹窗直接开始。"""
        # 0.7.6 P-1 拆分：预检对话框装配随 _on_preflight_done 迁入
        # panels/train_page.py，patch 目标随迁
        import xccv.training.training_window as tw
        import xccv.training.panels.train_page as tp

        pw = self._window()
        result = {"issues": [HealthIssue(LEVEL_INFO, "val_zero", "验证集比例为 0")],
                  "findings": [], "n_images": 4, "classes": ["cat", "dog"]}
        with mock.patch.object(tw.ProjectWindow, "_begin_training") as begin, \
                mock.patch.object(tp, "HealthReportDialog") as dlg_cls, \
                mock.patch.object(tp, "load_options",
                                  return_value=DevOptions(preflight_info_popup=False)):
            pw._on_preflight_done(result)
            begin.assert_called_once()
            dlg_cls.assert_not_called()

    def test_info_only_preflight_pops_when_enabled(self):
        """默认开启"提示级弹窗"：仅提示也弹预检窗。"""
        import xccv.training.training_window as tw
        import xccv.training.panels.train_page as tp

        pw = self._window()
        result = {"issues": [HealthIssue(LEVEL_INFO, "val_zero", "验证集比例为 0")],
                  "findings": [], "n_images": 4, "classes": ["cat", "dog"]}
        with mock.patch.object(tw.ProjectWindow, "_begin_training") as begin, \
                mock.patch.object(tp, "HealthReportDialog") as dlg_cls:
            dlg_cls.return_value.exec_.return_value = tw.QDialog.Accepted
            pw._on_preflight_done(result)
            dlg_cls.assert_called_once()
            begin.assert_called_once()

    def test_warn_preflight_shows_dialog_then_starts(self):
        import xccv.training.training_window as tw
        import xccv.training.panels.train_page as tp

        pw = self._window()
        result = {"issues": [HealthIssue(LEVEL_WARN, "imbalance", "类别不均衡")],
                  "findings": [Finding(LEVEL_WARN, "patience", "早停不触发",
                                       field="patience", value=5)],
                  "n_images": 4, "classes": ["cat", "dog"]}
        with mock.patch.object(tw.ProjectWindow, "_begin_training") as begin, \
                mock.patch.object(tp, "HealthReportDialog") as dlg_cls:
            dlg_cls.return_value.exec_.return_value = tw.QDialog.Accepted
            pw._on_preflight_done(result)
            dlg_cls.assert_called_once()
            begin.assert_called_once()


    def test_preflight_worker_runs_end_to_end(self):
        """后台线程同步执行：体检 + 参数建议一起产出（不弹窗）。"""
        from xccv.training.training_window import _PreflightWorker

        pw = self._window()
        results = []
        worker = _PreflightWorker(pw.project)
        worker.done.connect(results.append)
        worker.run()  # 不 start()：在调用线程内同步执行
        self.assertEqual(len(results), 1)
        payload = results[0]
        self.assertEqual(payload["n_images"], 4)
        self.assertEqual(payload["classes"], ["cat", "dog"])
        self.assertIsInstance(payload["issues"], list)
        self.assertIsInstance(payload["findings"], list)


class CompareEntryTest(_TempEnv):
    def _window_with_runs(self):
        from xccv.training.runs import RunRecord
        from xccv.training.training_window import ProjectWindow

        pw = ProjectWindow(load_project("ui"))
        self.addCleanup(pw.close)
        recs = []
        for i in range(3):
            run = Path(self._tmp.name) / f"run{i}"
            run.mkdir()
            rec = RunRecord(run_dir=run, status="completed",
                            start_time=f"2026-09-1{i} 10:00:00",
                            metrics={"config": {"model_name": "yolo11n"},
                                     "history": [{"epoch": 1, "train_loss": 2.0}]})
            recs.append(rec)
        return pw, recs

    def test_selection_capped_at_three(self):
        """多选上限 3 条：第 4 个选择被拒绝并回退。"""
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QListWidgetItem

        pw, recs = self._window_with_runs()
        self.assertTrue(pw.btn_compare.isEnabled())  # 按钮常启用，点击时校验
        for rec in recs:
            item = QListWidgetItem(str(rec.run_dir))
            item.setData(Qt.UserRole, str(rec.run_dir))   # 回退按键（与真实列表一致）
            item.setData(Qt.UserRole + 1, rec)
            pw.run_list.addItem(item)
        item4 = QListWidgetItem("extra")
        item4.setData(Qt.UserRole, "extra")
        item4.setData(Qt.UserRole + 1, recs[0])
        pw.run_list.addItem(item4)
        for it in tuple(pw.run_list.item(i) for i in range(4)):
            it.setSelected(True)
        self.assertEqual(len(pw.run_list.selectedItems()), 3)  # 第 4 个被拒
        self.assertFalse(item4.isSelected())

    def test_open_compare_with_stub(self):
        import xccv.training.compare_window as cw
        import xccv.training.training_window as tw
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QListWidgetItem

        pw, recs = self._window_with_runs()
        for rec in recs[:2]:
            item = QListWidgetItem(str(rec.run_dir))
            item.setData(Qt.UserRole + 1, rec)
            pw.run_list.addItem(item)
            item.setSelected(True)  # 先入列再选中（脱离列表的条目选中无效）
        with mock.patch.object(cw, "CompareWindow") as cmp_cls:
            pw._open_compare()
            cmp_cls.assert_called_once()
            self.assertIsNotNone(pw._compare_window)


class LauncherArgTest(unittest.TestCase):
    def test_launch_with_arg_appends(self):
        import xccv.launcher as launcher

        captured = {}
        with mock.patch.object(launcher, "is_installed", return_value=False), \
                mock.patch.object(launcher.subprocess, "Popen",
                                  side_effect=lambda cmd, **kw:
                                  captured.update(cmd=cmd)):
            launcher.launch_platform("annotation", "D:/datasets/demo")
        self.assertEqual(captured["cmd"][-1], "D:/datasets/demo")
        self.assertIn("-m", captured["cmd"])

    def test_launch_without_arg(self):
        import xccv.launcher as launcher

        captured = {}
        with mock.patch.object(launcher, "is_installed", return_value=False), \
                mock.patch.object(launcher.subprocess, "Popen",
                                  side_effect=lambda cmd, **kw:
                                  captured.update(cmd=cmd)):
            launcher.launch_platform("annotation")
        self.assertEqual(len(captured["cmd"]), 3)  # pythonw -m module


class FontSizeAuditTest(unittest.TestCase):
    def test_no_font_below_14px(self):
        """全局字号审查（界面可读性约束）：QSS 中不得低于 14px。"""
        import re

        from xccv.ui_style import APP_QSS

        sizes = [int(m) for m in re.findall(r"font-size:\s*(\d+)px", APP_QSS)]
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), 15, f"过小字号出现：{sizes}")

if __name__ == '__main__':
    unittest.main()
