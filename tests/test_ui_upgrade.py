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

        self.assertEqual(xccv.__version__, "0.8.2")

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


if __name__ == "__main__":
    unittest.main()



